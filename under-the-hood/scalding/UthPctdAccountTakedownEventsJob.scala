package com.twitter.visibility.under_the_hood

import com.twitter.common_internal.analytics.test_user_filter.TestUserFilter
import com.twitter.guano.thriftscala.PctdAction
import com.twitter.guano.thriftscala.PctdActionType
import com.twitter.scalding.Args
import com.twitter.scalding.DateOps
import com.twitter.scalding.DateParser
import com.twitter.scalding.DateRange
import com.twitter.scalding.Days
import com.twitter.scalding.Execution
import com.twitter.scalding.TypedPipe
import com.twitter.scalding.parquet.scrooge.PartitionedParquetScroogeSource
import com.twitter.scalding_internal.dalv2.DAL
import com.twitter.scalding_internal.dalv2.remote_access.AllowCrossDC
import com.twitter.scalding_internal.job.TwitterExecutionApp
import com.twitter.tseng.withholding.thriftscala.TakedownReason
import com.twitter.twadoop.user.gen.thriftscala.CombinedUser
import com.twitter.usersource.snapshot.combined.UsersourceScalaDataset
import com.twitter.visibility.under_the_hood.events.thriftscala.UthAccountTakedownEvent
import java.time.Instant
import java.time.ZoneOffset
import java.util.Locale
import java.util.TimeZone
import twadoop_config.configuration.log_categories.group.useng.AuditServicePctdActionsScalaDataset

class UthPctdAccountTakedownEventsApp {
  import UthPctdAccountTakedownEventsApp._

  implicit val tz: TimeZone = DateOps.UTC

  def runOnDateRange(
    dateRange: DateRange,
    config: UthPctdAccountTakedownEventsConfig
  ): Execution[Unit] = {
    val ingestionRange = DateRange(dateRange.start, dateRange.end + Days(1))
    val pctdActions = DAL.read(AuditServicePctdActionsScalaDataset, ingestionRange).toTypedPipe
    val usersourceUsers = DAL
      .readMostRecentSnapshot(
        UsersourceScalaDataset,
        DateRange(dateRange.end - Days(config.usersourceSnapshotMaxAgeDays), dateRange.end)
      )
      .withRemoteReadPolicy(AllowCrossDC)
      .toTypedPipe

    withheldEventDays(pctdActions, usersourceUsers, dateRange, config.testUserIds, config.reducers)
      .map {
        case ((userId, (reason, country)), day) =>
          val partition = f"${day / 10000}%04d/${day / 100 % 100}%02d/${day % 100}%02d"
          (partition, UthAccountTakedownEvent(Some(userId), Some(day), Some(reason), Some(country)))
      }
      .writeExecution(
        PartitionedParquetScroogeSource[String, UthAccountTakedownEvent](
          s"${config.outputPath}/daily_account_takedowns",
          "%s"))
  }
}

object UthPctdAccountTakedownEventsApp {
  private val ReasonPriority = Map(
    "LEGAL_REQUEST" -> 0,
    "BYSTANDER_REPORT" -> 1,
    "DMCA" -> 2,
    "UNSPECIFIED" -> 3
  )

  private[under_the_hood] def withheldEventDays(
    pctdActions: TypedPipe[PctdAction],
    usersourceUsers: TypedPipe[CombinedUser],
    dateRange: DateRange,
    testUserIds: Set[Long],
    reducers: Int
  ): TypedPipe[((Long, (String, String)), Int)] = {
    val eventDays = pctdActions
      .filter { a =>
        a.`type` == PctdActionType.User && a.takendown && a.countryCode.nonEmpty &&
        !TestUserFilter.isTestUserId(a.userId) &&
        (testUserIds.isEmpty || testUserIds(a.userId)) &&
        a.timestamp.toLong * 1000L >= dateRange.start.timestamp &&
        a.timestamp.toLong * 1000L <= dateRange.end.timestamp
      }
      .map { a =>
        val day = Instant.ofEpochSecond(a.timestamp.toLong).atZone(ZoneOffset.UTC).toLocalDate
        (
          (a.userId, a.countryCode.toLowerCase(Locale.ROOT)),
          day.getYear * 10000 + day.getMonthValue * 100 + day.getDayOfMonth
        )
      }
      .distinct
    val reasons = usersourceUsers
      .flatMap { cu =>
        for {
          u <- cu.user.toSeq
          if testUserIds.isEmpty || testUserIds(u.id)
          t <- u.takedowns.toSeq
          r <- t.takedownCountryReasons.toSeq.flatten
          pair <- (r.takedownReason match {
              case TakedownReason.LegalRequest(v) => Some((v.countryCode, "LEGAL_REQUEST"))
              case TakedownReason.BystanderReport(v) => Some((v.countryCode, "BYSTANDER_REPORT"))
              case TakedownReason.Dmca(_) => Some(("xy", "DMCA"))
              case TakedownReason.UnspecifiedReason(v) => Some((v.countryCode, "UNSPECIFIED"))
              case _ => None
            })
        } yield ((u.id, pair._1.toLowerCase(Locale.ROOT)), pair._2)
      }.group.reduce { (a, b) => if (ReasonPriority(a) <= ReasonPriority(b)) a else b }

    val grouped = eventDays.group
    val partitioned = if (reducers > 0) grouped.withReducers(reducers) else grouped
    partitioned.leftJoin(reasons).toTypedPipe.map {
      case ((userId, country), (day, reason)) =>
        ((userId, (reason.getOrElse("UNSPECIFIED"), country)), day)
    }
  }
}

case class UthPctdAccountTakedownEventsConfig(
  testUserIds: Set[Long],
  reducers: Int,
  usersourceSnapshotMaxAgeDays: Int,
  outputPath: String)

object UthPctdAccountTakedownEventsConfig {
  def fromArgs(args: Args): UthPctdAccountTakedownEventsConfig =
    UthPctdAccountTakedownEventsConfig(
      testUserIds = args
        .list("userIds")
        .flatMap(_.split(",")).map(_.trim).filter(_.nonEmpty).map(_.toLong).toSet,
      reducers = args.int("reducers", 50),
      usersourceSnapshotMaxAgeDays = args.int("usersourceSnapshotMaxAgeDays", 10),
      outputPath = args.required("outputPath")
    )
}

object UthPctdAccountTakedownEventsAdhoc
    extends UthPctdAccountTakedownEventsApp
    with TwitterExecutionApp {
  override def job: Execution[Unit] = Execution.withArgs { args =>
    implicit val dp: DateParser = DateParser.default
    runOnDateRange(
      DateRange.parse(args.list("date")),
      UthPctdAccountTakedownEventsConfig.fromArgs(args))
  }
}
