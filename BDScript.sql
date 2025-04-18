USE [master]
GO
/****** Object:  Database [PersonInfoForFaceRec]    Script Date: 07.04.2025 17:35:30 ******/
CREATE DATABASE [PersonInfoForFaceRec]
 CONTAINMENT = NONE
 ON  PRIMARY 
( NAME = N'PersonInfoForFaceRec', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL15.SQLEXPRESS\MSSQL\DATA\PersonInfoForFaceRec.mdf' , SIZE = 8192KB , MAXSIZE = UNLIMITED, FILEGROWTH = 65536KB )
 LOG ON 
( NAME = N'PersonInfoForFaceRec_log', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL15.SQLEXPRESS\MSSQL\DATA\PersonInfoForFaceRec_log.ldf' , SIZE = 8192KB , MAXSIZE = 2048GB , FILEGROWTH = 65536KB )
GO
IF (1 = FULLTEXTSERVICEPROPERTY('IsFullTextInstalled'))
begin
EXEC [PersonInfoForFaceRec].[dbo].[sp_fulltext_database] @action = 'enable'
end
GO
ALTER DATABASE [PersonInfoForFaceRec] SET ANSI_NULL_DEFAULT OFF 
GO
ALTER DATABASE [PersonInfoForFaceRec] SET ANSI_NULLS OFF 
GO
ALTER DATABASE [PersonInfoForFaceRec] SET ANSI_PADDING OFF 
GO
ALTER DATABASE [PersonInfoForFaceRec] SET ANSI_WARNINGS OFF 
GO
ALTER DATABASE [PersonInfoForFaceRec] SET ARITHABORT OFF 
GO
ALTER DATABASE [PersonInfoForFaceRec] SET AUTO_CLOSE OFF 
GO
ALTER DATABASE [PersonInfoForFaceRec] SET AUTO_SHRINK OFF 
GO
ALTER DATABASE [PersonInfoForFaceRec] SET AUTO_UPDATE_STATISTICS ON 
GO
ALTER DATABASE [PersonInfoForFaceRec] SET CURSOR_CLOSE_ON_COMMIT OFF 
GO
ALTER DATABASE [PersonInfoForFaceRec] SET CURSOR_DEFAULT  GLOBAL 
GO
ALTER DATABASE [PersonInfoForFaceRec] SET CONCAT_NULL_YIELDS_NULL OFF 
GO
ALTER DATABASE [PersonInfoForFaceRec] SET NUMERIC_ROUNDABORT OFF 
GO
ALTER DATABASE [PersonInfoForFaceRec] SET QUOTED_IDENTIFIER OFF 
GO
ALTER DATABASE [PersonInfoForFaceRec] SET RECURSIVE_TRIGGERS OFF 
GO
ALTER DATABASE [PersonInfoForFaceRec] SET  DISABLE_BROKER 
GO
ALTER DATABASE [PersonInfoForFaceRec] SET AUTO_UPDATE_STATISTICS_ASYNC OFF 
GO
ALTER DATABASE [PersonInfoForFaceRec] SET DATE_CORRELATION_OPTIMIZATION OFF 
GO
ALTER DATABASE [PersonInfoForFaceRec] SET TRUSTWORTHY OFF 
GO
ALTER DATABASE [PersonInfoForFaceRec] SET ALLOW_SNAPSHOT_ISOLATION OFF 
GO
ALTER DATABASE [PersonInfoForFaceRec] SET PARAMETERIZATION SIMPLE 
GO
ALTER DATABASE [PersonInfoForFaceRec] SET READ_COMMITTED_SNAPSHOT OFF 
GO
ALTER DATABASE [PersonInfoForFaceRec] SET HONOR_BROKER_PRIORITY OFF 
GO
ALTER DATABASE [PersonInfoForFaceRec] SET RECOVERY SIMPLE 
GO
ALTER DATABASE [PersonInfoForFaceRec] SET  MULTI_USER 
GO
ALTER DATABASE [PersonInfoForFaceRec] SET PAGE_VERIFY CHECKSUM  
GO
ALTER DATABASE [PersonInfoForFaceRec] SET DB_CHAINING OFF 
GO
ALTER DATABASE [PersonInfoForFaceRec] SET FILESTREAM( NON_TRANSACTED_ACCESS = OFF ) 
GO
ALTER DATABASE [PersonInfoForFaceRec] SET TARGET_RECOVERY_TIME = 60 SECONDS 
GO
ALTER DATABASE [PersonInfoForFaceRec] SET DELAYED_DURABILITY = DISABLED 
GO
USE [PersonInfoForFaceRec]
GO
/****** Object:  Table [dbo].[PersonInfo]    Script Date: 07.04.2025 17:35:30 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[PersonInfo](
	[ID] [int] NOT NULL,
	[LastName] [nvarchar](max) NOT NULL,
	[FirstName] [nvarchar](50) NOT NULL,
	[Patronymic] [nvarchar](50) NULL,
	[Birthday] [date] NOT NULL,
	[Activity] [nvarchar](50) NOT NULL,
 CONSTRAINT [PK_PersonInfo] PRIMARY KEY CLUSTERED 
(
	[ID] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON) ON [PRIMARY]
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
INSERT [dbo].[PersonInfo] ([ID], [LastName], [FirstName], [Patronymic], [Birthday], [Activity]) VALUES (0, N'Anvarov', N'Niaz', N'Nailevich', CAST(N'2006-01-22' AS Date), N'Student')
INSERT [dbo].[PersonInfo] ([ID], [LastName], [FirstName], [Patronymic], [Birthday], [Activity]) VALUES (1, N'Urazbaev', N'Gazinur', N'Rishatovich', CAST(N'2005-08-23' AS Date), N'Spectravod')
INSERT [dbo].[PersonInfo] ([ID], [LastName], [FirstName], [Patronymic], [Birthday], [Activity]) VALUES (2, N'Timerzyanov', N'Rishat', N'Airatovich', CAST(N'1952-10-07' AS Date), N'Vladelets nedvizhimosti')
GO
USE [master]
GO
ALTER DATABASE [PersonInfoForFaceRec] SET  READ_WRITE 
GO
